#pragma once

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace dotnet_gui {
	using namespace System::Diagnostics;

	/// <summary>
	/// Zusammenfassung für UpdateForm
	///
	/// Warnung: Wenn Sie den Namen dieser Klasse ändern, müssen Sie auch
	///          die Ressourcendateiname-Eigenschaft für das Tool zur Kompilierung verwalteter Ressourcen ändern,
	///          das allen RESX-Dateien zugewiesen ist, von denen diese Klasse abhängt.
	///          Anderenfalls können die Designer nicht korrekt mit den lokalisierten Ressourcen
	///          arbeiten, die diesem Formular zugewiesen sind.
	/// </summary>
	public ref class UpdateForm : public System::Windows::Forms::Form
	{
	public:
		UpdateForm(void)
		{
			InitializeComponent();
			//
			//TODO: Konstruktorcode hier hinzufügen.
			//
		}

	protected:
		/// <summary>
		/// Verwendete Ressourcen bereinigen.
		/// </summary>
		~UpdateForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Label^  label1;
	public: System::Windows::Forms::Label^  labelVersion;
	private: 

	protected: 

	private: System::Windows::Forms::Label^  label3;
	public: System::Windows::Forms::TextBox^  textChanges;
	private: 


	private: System::Windows::Forms::Label^  label4;
	public: System::Windows::Forms::Button^  bnYes;
	private: 

	private: System::Windows::Forms::Button^  bnNo;
	public: System::Windows::Forms::CheckBox^  checkDontAsk;
	private: 





	private:
		/// <summary>
		/// Erforderliche Designervariable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Erforderliche Methode für die Designerunterstützung.
		/// Der Inhalt der Methode darf nicht mit dem Code-Editor geändert werden.
		/// </summary>
		void InitializeComponent(void)
		{
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->labelVersion = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->textChanges = (gcnew System::Windows::Forms::TextBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->bnYes = (gcnew System::Windows::Forms::Button());
			this->bnNo = (gcnew System::Windows::Forms::Button());
			this->checkDontAsk = (gcnew System::Windows::Forms::CheckBox());
			this->SuspendLayout();
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(14, 11);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(76, 13);
			this->label1->TabIndex = 3;
			this->label1->Text = L"Latest version:";
			// 
			// labelVersion
			// 
			this->labelVersion->AutoSize = true;
			this->labelVersion->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->labelVersion->Location = System::Drawing::Point(93, 11);
			this->labelVersion->Name = L"labelVersion";
			this->labelVersion->Size = System::Drawing::Size(41, 13);
			this->labelVersion->TabIndex = 4;
			this->labelVersion->Text = L"label2";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(13, 41);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(124, 13);
			this->label3->TabIndex = 5;
			this->label3->Text = L"Changes / new features:";
			// 
			// textChanges
			// 
			this->textChanges->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
				| System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->textChanges->BackColor = System::Drawing::Color::White;
			this->textChanges->Location = System::Drawing::Point(17, 64);
			this->textChanges->Multiline = true;
			this->textChanges->Name = L"textChanges";
			this->textChanges->ReadOnly = true;
			this->textChanges->ScrollBars = System::Windows::Forms::ScrollBars::Both;
			this->textChanges->Size = System::Drawing::Size(264, 114);
			this->textChanges->TabIndex = 6;
			// 
			// label4
			// 
			this->label4->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(14, 194);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(108, 13);
			this->label4->TabIndex = 7;
			this->label4->Text = L"Visit download page\?";
			// 
			// bnYes
			// 
			this->bnYes->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->bnYes->Location = System::Drawing::Point(17, 220);
			this->bnYes->Name = L"bnYes";
			this->bnYes->Size = System::Drawing::Size(93, 26);
			this->bnYes->TabIndex = 0;
			this->bnYes->Text = L"Yes";
			this->bnYes->UseVisualStyleBackColor = true;
			this->bnYes->Click += gcnew System::EventHandler(this, &UpdateForm::bnYes_Click);
			// 
			// bnNo
			// 
			this->bnNo->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->bnNo->Location = System::Drawing::Point(127, 220);
			this->bnNo->Name = L"bnNo";
			this->bnNo->Size = System::Drawing::Size(93, 26);
			this->bnNo->TabIndex = 1;
			this->bnNo->Text = L"No";
			this->bnNo->UseVisualStyleBackColor = true;
			this->bnNo->Click += gcnew System::EventHandler(this, &UpdateForm::bnNo_Click);
			// 
			// checkDontAsk
			// 
			this->checkDontAsk->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->checkDontAsk->AutoSize = true;
			this->checkDontAsk->Location = System::Drawing::Point(16, 255);
			this->checkDontAsk->Name = L"checkDontAsk";
			this->checkDontAsk->Size = System::Drawing::Size(117, 17);
			this->checkDontAsk->TabIndex = 2;
			this->checkDontAsk->Text = L"Don\'t ask me again";
			this->checkDontAsk->UseVisualStyleBackColor = true;
			// 
			// UpdateForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(301, 281);
			this->Controls->Add(this->checkDontAsk);
			this->Controls->Add(this->bnNo);
			this->Controls->Add(this->bnYes);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->textChanges);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->labelVersion);
			this->Controls->Add(this->label1);
			this->MinimumSize = System::Drawing::Size(242, 241);
			this->Name = L"UpdateForm";
			this->Text = L"New version of CUJ2K available";
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

		public: String^ downloadPage;

	private: System::Void bnYes_Click(System::Object^  sender, System::EventArgs^  e) {
				 Process::Start(downloadPage);
				 Visible = false;
				 //this->Close();
			 }
private: System::Void bnNo_Click(System::Object^  sender, System::EventArgs^  e) {
			 //this->Close();
			 Visible = false;
		 }
};
}
