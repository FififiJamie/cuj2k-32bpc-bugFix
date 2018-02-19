#pragma once

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace dotnet_gui {

	/// <summary>
	/// Zusammenfassung für AdvancedForm
	///
	/// Warnung: Wenn Sie den Namen dieser Klasse ändern, müssen Sie auch
	///          die Ressourcendateiname-Eigenschaft für das Tool zur Kompilierung verwalteter Ressourcen ändern,
	///          das allen RESX-Dateien zugewiesen ist, von denen diese Klasse abhängt.
	///          Anderenfalls können die Designer nicht korrekt mit den lokalisierten Ressourcen
	///          arbeiten, die diesem Formular zugewiesen sind.
	/// </summary>
	public ref class AdvancedForm : public System::Windows::Forms::Form
	{
	public:
		AdvancedForm(void)
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
		~AdvancedForm()
		{
			if (components)
			{
				delete components;
			}
		}
	public: System::Windows::Forms::TextBox^  textAddOptions;
	protected: 

	protected: 
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::GroupBox^  groupBox2;
	public: System::Windows::Forms::ComboBox^  comboCUDAdevices;
	private: 
	public: System::Windows::Forms::CheckBox^  checkBenchmark;
	public: System::Windows::Forms::CheckBox^  checkStreaming;



	private: System::Windows::Forms::Label^  label4;
	public: System::Windows::Forms::ComboBox^  comboCBSize;
	private: 

	private: System::Windows::Forms::Label^  label3;
	public: System::Windows::Forms::CheckBox^  checkUpdates;
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
			this->textAddOptions = (gcnew System::Windows::Forms::TextBox());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->comboCUDAdevices = (gcnew System::Windows::Forms::ComboBox());
			this->checkBenchmark = (gcnew System::Windows::Forms::CheckBox());
			this->checkStreaming = (gcnew System::Windows::Forms::CheckBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->comboCBSize = (gcnew System::Windows::Forms::ComboBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->checkUpdates = (gcnew System::Windows::Forms::CheckBox());
			this->groupBox2->SuspendLayout();
			this->SuspendLayout();
			// 
			// textAddOptions
			// 
			this->textAddOptions->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->textAddOptions->Location = System::Drawing::Point(122, 161);
			this->textAddOptions->Name = L"textAddOptions";
			this->textAddOptions->Size = System::Drawing::Size(172, 20);
			this->textAddOptions->TabIndex = 20;
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(24, 164);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(92, 13);
			this->label5->TabIndex = 19;
			this->label5->Text = L"additional options:";
			// 
			// groupBox2
			// 
			this->groupBox2->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->groupBox2->Controls->Add(this->comboCUDAdevices);
			this->groupBox2->Controls->Add(this->checkBenchmark);
			this->groupBox2->Controls->Add(this->checkStreaming);
			this->groupBox2->Controls->Add(this->label4);
			this->groupBox2->Location = System::Drawing::Point(10, 12);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(298, 112);
			this->groupBox2->TabIndex = 18;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"CUDA";
			// 
			// comboCUDAdevices
			// 
			this->comboCUDAdevices->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->comboCUDAdevices->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->comboCUDAdevices->FormattingEnabled = true;
			this->comboCUDAdevices->Items->AddRange(gcnew cli::array< System::Object^  >(1) {L"(auto-select)"});
			this->comboCUDAdevices->Location = System::Drawing::Point(60, 18);
			this->comboCUDAdevices->Name = L"comboCUDAdevices";
			this->comboCUDAdevices->Size = System::Drawing::Size(224, 21);
			this->comboCUDAdevices->TabIndex = 7;
			// 
			// checkBenchmark
			// 
			this->checkBenchmark->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->checkBenchmark->CheckAlign = System::Drawing::ContentAlignment::TopLeft;
			this->checkBenchmark->Location = System::Drawing::Point(17, 75);
			this->checkBenchmark->Name = L"checkBenchmark";
			this->checkBenchmark->Size = System::Drawing::Size(269, 30);
			this->checkBenchmark->TabIndex = 6;
			this->checkBenchmark->Text = L"benchmark (only time measurement, no JPEG2000 files are generated)";
			this->checkBenchmark->TextAlign = System::Drawing::ContentAlignment::TopLeft;
			this->checkBenchmark->UseVisualStyleBackColor = true;
			// 
			// checkStreaming
			// 
			this->checkStreaming->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->checkStreaming->CheckAlign = System::Drawing::ContentAlignment::TopLeft;
			this->checkStreaming->Checked = true;
			this->checkStreaming->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkStreaming->Location = System::Drawing::Point(17, 45);
			this->checkStreaming->Name = L"checkStreaming";
			this->checkStreaming->Size = System::Drawing::Size(267, 30);
			this->checkStreaming->TabIndex = 5;
			this->checkStreaming->Text = L"enable streaming (speedup for large pictures on high-end GPUs)";
			this->checkStreaming->TextAlign = System::Drawing::ContentAlignment::TopLeft;
			this->checkStreaming->UseVisualStyleBackColor = true;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(12, 21);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(42, 13);
			this->label4->TabIndex = 0;
			this->label4->Text = L"device:";
			// 
			// comboCBSize
			// 
			this->comboCBSize->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->comboCBSize->FormattingEnabled = true;
			this->comboCBSize->Items->AddRange(gcnew cli::array< System::Object^  >(4) {L"(auto)", L"16x16", L"32x32", L"64x64"});
			this->comboCBSize->Location = System::Drawing::Point(122, 134);
			this->comboCBSize->Name = L"comboCBSize";
			this->comboCBSize->Size = System::Drawing::Size(75, 21);
			this->comboCBSize->TabIndex = 17;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(24, 137);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(81, 13);
			this->label3->TabIndex = 16;
			this->label3->Text = L"codeblock size:";
			// 
			// checkUpdates
			// 
			this->checkUpdates->AutoSize = true;
			this->checkUpdates->Location = System::Drawing::Point(27, 191);
			this->checkUpdates->Name = L"checkUpdates";
			this->checkUpdates->Size = System::Drawing::Size(143, 17);
			this->checkUpdates->TabIndex = 21;
			this->checkUpdates->Text = L"check online for updates";
			this->checkUpdates->UseVisualStyleBackColor = true;
			// 
			// AdvancedForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(135)), static_cast<System::Int32>(static_cast<System::Byte>(174)), 
				static_cast<System::Int32>(static_cast<System::Byte>(197)));
			this->ClientSize = System::Drawing::Size(320, 222);
			this->Controls->Add(this->checkUpdates);
			this->Controls->Add(this->textAddOptions);
			this->Controls->Add(this->label5);
			this->Controls->Add(this->groupBox2);
			this->Controls->Add(this->comboCBSize);
			this->Controls->Add(this->label3);
			this->MinimumSize = System::Drawing::Size(272, 249);
			this->Name = L"AdvancedForm";
			this->Text = L"Advanced options";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &AdvancedForm::AdvancedForm_FormClosing);
			this->groupBox2->ResumeLayout(false);
			this->groupBox2->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void AdvancedForm_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e) {
				 e->Cancel = true;
				 Hide();
			 }
};
}
