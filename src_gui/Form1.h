#pragma once

#include <cutil.h>
#include <cutil_inline.h>
#include "HttpHelper.h"
#include "UpdateForm.h"
#include "AdvancedForm.h"

namespace dotnet_gui {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Text;
	using namespace System::Diagnostics;
	using namespace System::Globalization;
	using namespace System::Threading;

	/// <summary>
	/// Zusammenfassung für Form1
	///
	/// Warnung: Wenn Sie den Namen dieser Klasse ändern, müssen Sie auch
	///          die Ressourcendateiname-Eigenschaft für das Tool zur Kompilierung verwalteter Ressourcen ändern,
	///          das allen RESX-Dateien zugewiesen ist, von denen diese Klasse abhängt.
	///          Anderenfalls können die Designer nicht korrekt mit den lokalisierten Ressourcen
	///          arbeiten, die diesem Formular zugewiesen sind.
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:
		Form1(void)
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
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::PictureBox^  pictureBox1;
	private: System::Windows::Forms::LinkLabel^  linkLabel1;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::ListBox^  inputfilesList;

	private: System::Windows::Forms::Button^  addBn;
	private: System::Windows::Forms::Button^  removeBn;
	private: System::Windows::Forms::Button^  removeAllBn;
	private: System::Windows::Forms::Button^  encodeBn;




	private: System::Windows::Forms::OpenFileDialog^  inputfilesDlg;
	private: System::Windows::Forms::RadioButton^  radioLossless;
	private: System::Windows::Forms::RadioButton^  radioLossy;
	private: System::Windows::Forms::CheckBox^  checkHQ;
	private: System::Windows::Forms::ComboBox^  comboSizeMode;





	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::TextBox^  textOutputSize;















	private: System::Windows::Forms::GroupBox^  groupBox1;






	private: System::Windows::Forms::LinkLabel^  linkLabel2;
	private: System::Windows::Forms::Label^  label6;

	private: System::Windows::Forms::Panel^  panel1;

	private: System::Windows::Forms::Label^  label7;
	private: System::Windows::Forms::ComboBox^  comboOutFormat;

	private: System::Windows::Forms::Label^  label8;
	private: System::Windows::Forms::Button^  advancedBn;
	private: System::Windows::Forms::Button^  resetBn;


	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::TextBox^  textMJ2Output;
	private: System::Windows::Forms::Button^  MJ2selectBn;
	private: System::Windows::Forms::SaveFileDialog^  MJ2outputDlg;







	protected: 

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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(Form1::typeid));
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->linkLabel1 = (gcnew System::Windows::Forms::LinkLabel());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->inputfilesList = (gcnew System::Windows::Forms::ListBox());
			this->addBn = (gcnew System::Windows::Forms::Button());
			this->removeBn = (gcnew System::Windows::Forms::Button());
			this->removeAllBn = (gcnew System::Windows::Forms::Button());
			this->encodeBn = (gcnew System::Windows::Forms::Button());
			this->inputfilesDlg = (gcnew System::Windows::Forms::OpenFileDialog());
			this->radioLossless = (gcnew System::Windows::Forms::RadioButton());
			this->radioLossy = (gcnew System::Windows::Forms::RadioButton());
			this->checkHQ = (gcnew System::Windows::Forms::CheckBox());
			this->comboSizeMode = (gcnew System::Windows::Forms::ComboBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->textOutputSize = (gcnew System::Windows::Forms::TextBox());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->linkLabel2 = (gcnew System::Windows::Forms::LinkLabel());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->comboOutFormat = (gcnew System::Windows::Forms::ComboBox());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->advancedBn = (gcnew System::Windows::Forms::Button());
			this->resetBn = (gcnew System::Windows::Forms::Button());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->textMJ2Output = (gcnew System::Windows::Forms::TextBox());
			this->MJ2selectBn = (gcnew System::Windows::Forms::Button());
			this->MJ2outputDlg = (gcnew System::Windows::Forms::SaveFileDialog());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->pictureBox1))->BeginInit();
			this->groupBox1->SuspendLayout();
			this->panel1->SuspendLayout();
			this->SuspendLayout();
			// 
			// pictureBox1
			// 
			this->pictureBox1->BackgroundImage = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"pictureBox1.BackgroundImage")));
			this->pictureBox1->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"pictureBox1.Image")));
			this->pictureBox1->Location = System::Drawing::Point(-2, -11);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(133, 100);
			this->pictureBox1->TabIndex = 0;
			this->pictureBox1->TabStop = false;
			// 
			// linkLabel1
			// 
			this->linkLabel1->ActiveLinkColor = System::Drawing::Color::DarkRed;
			this->linkLabel1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->linkLabel1->AutoSize = true;
			this->linkLabel1->BackColor = System::Drawing::Color::Transparent;
			this->linkLabel1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->linkLabel1->LinkColor = System::Drawing::Color::DarkRed;
			this->linkLabel1->Location = System::Drawing::Point(141, 65);
			this->linkLabel1->Name = L"linkLabel1";
			this->linkLabel1->Size = System::Drawing::Size(175, 13);
			this->linkLabel1->TabIndex = 0;
			this->linkLabel1->TabStop = true;
			this->linkLabel1->Text = L"http://cuj2k.sourceforge.net/";
			this->linkLabel1->VisitedLinkColor = System::Drawing::Color::DarkRed;
			this->linkLabel1->LinkClicked += gcnew System::Windows::Forms::LinkLabelLinkClickedEventHandler(this, &Form1::linkLabel1_LinkClicked);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(12, 108);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(55, 13);
			this->label1->TabIndex = 2;
			this->label1->Text = L"Input files:";
			// 
			// inputfilesList
			// 
			this->inputfilesList->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
				| System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->inputfilesList->FormattingEnabled = true;
			this->inputfilesList->HorizontalScrollbar = true;
			this->inputfilesList->Location = System::Drawing::Point(73, 108);
			this->inputfilesList->Name = L"inputfilesList";
			this->inputfilesList->SelectionMode = System::Windows::Forms::SelectionMode::MultiExtended;
			this->inputfilesList->Size = System::Drawing::Size(296, 147);
			this->inputfilesList->TabIndex = 3;
			this->inputfilesList->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &Form1::inputfilesList_KeyDown);
			// 
			// addBn
			// 
			this->addBn->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->addBn->Location = System::Drawing::Point(375, 108);
			this->addBn->Name = L"addBn";
			this->addBn->Size = System::Drawing::Size(71, 25);
			this->addBn->TabIndex = 4;
			this->addBn->Text = L"Add...";
			this->addBn->UseVisualStyleBackColor = true;
			this->addBn->Click += gcnew System::EventHandler(this, &Form1::button1_Click);
			// 
			// removeBn
			// 
			this->removeBn->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->removeBn->Location = System::Drawing::Point(375, 148);
			this->removeBn->Name = L"removeBn";
			this->removeBn->Size = System::Drawing::Size(71, 25);
			this->removeBn->TabIndex = 5;
			this->removeBn->Text = L"Remove";
			this->removeBn->UseVisualStyleBackColor = true;
			this->removeBn->Click += gcnew System::EventHandler(this, &Form1::removeBn_Click);
			// 
			// removeAllBn
			// 
			this->removeAllBn->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->removeAllBn->Location = System::Drawing::Point(375, 180);
			this->removeAllBn->Name = L"removeAllBn";
			this->removeAllBn->Size = System::Drawing::Size(71, 25);
			this->removeAllBn->TabIndex = 6;
			this->removeAllBn->Text = L"Remove all";
			this->removeAllBn->UseVisualStyleBackColor = true;
			this->removeAllBn->Click += gcnew System::EventHandler(this, &Form1::removeAllBn_Click);
			// 
			// encodeBn
			// 
			this->encodeBn->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->encodeBn->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->encodeBn->Location = System::Drawing::Point(72, 482);
			this->encodeBn->Name = L"encodeBn";
			this->encodeBn->Size = System::Drawing::Size(296, 40);
			this->encodeBn->TabIndex = 16;
			this->encodeBn->Text = L"Encode!";
			this->encodeBn->UseVisualStyleBackColor = true;
			this->encodeBn->Click += gcnew System::EventHandler(this, &Form1::encodeBn_Click);
			// 
			// inputfilesDlg
			// 
			this->inputfilesDlg->Filter = L"Bitmap-files (*.bmp)|*.bmp|All files (*.*)|*.*";
			this->inputfilesDlg->Multiselect = true;
			this->inputfilesDlg->RestoreDirectory = true;
			// 
			// radioLossless
			// 
			this->radioLossless->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->radioLossless->AutoSize = true;
			this->radioLossless->Location = System::Drawing::Point(13, 10);
			this->radioLossless->Name = L"radioLossless";
			this->radioLossless->Size = System::Drawing::Size(61, 17);
			this->radioLossless->TabIndex = 0;
			this->radioLossless->Text = L"lossless";
			this->radioLossless->UseVisualStyleBackColor = true;
			this->radioLossless->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioLossless_CheckedChanged);
			// 
			// radioLossy
			// 
			this->radioLossy->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->radioLossy->AutoSize = true;
			this->radioLossy->Checked = true;
			this->radioLossy->Location = System::Drawing::Point(13, 24);
			this->radioLossy->Name = L"radioLossy";
			this->radioLossy->Size = System::Drawing::Size(48, 17);
			this->radioLossy->TabIndex = 1;
			this->radioLossy->TabStop = true;
			this->radioLossy->Text = L"lossy";
			this->radioLossy->UseVisualStyleBackColor = true;
			this->radioLossy->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioLossy_CheckedChanged);
			// 
			// checkHQ
			// 
			this->checkHQ->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->checkHQ->AutoSize = true;
			this->checkHQ->Location = System::Drawing::Point(80, 24);
			this->checkHQ->Name = L"checkHQ";
			this->checkHQ->Size = System::Drawing::Size(239, 17);
			this->checkHQ->TabIndex = 2;
			this->checkHQ->Text = L"HQ (better quality, but bigger files and slower)";
			this->checkHQ->UseVisualStyleBackColor = true;
			// 
			// comboSizeMode
			// 
			this->comboSizeMode->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->comboSizeMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->comboSizeMode->FormattingEnabled = true;
			this->comboSizeMode->Items->AddRange(gcnew cli::array< System::Object^  >(6) {L"(maximum)", L"% of input file", L"compression ratio x:1", 
				L"bytes", L"KB", L"MB"});
			this->comboSizeMode->Location = System::Drawing::Point(241, 320);
			this->comboSizeMode->Name = L"comboSizeMode";
			this->comboSizeMode->Size = System::Drawing::Size(127, 21);
			this->comboSizeMode->TabIndex = 10;
			this->comboSizeMode->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboSizeMode_SelectedIndexChanged);
			// 
			// label2
			// 
			this->label2->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(70, 324);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(76, 13);
			this->label2->TabIndex = 8;
			this->label2->Text = L"Output filesize:";
			// 
			// textOutputSize
			// 
			this->textOutputSize->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->textOutputSize->Location = System::Drawing::Point(160, 321);
			this->textOutputSize->Name = L"textOutputSize";
			this->textOutputSize->Size = System::Drawing::Size(75, 20);
			this->textOutputSize->TabIndex = 9;
			// 
			// groupBox1
			// 
			this->groupBox1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->groupBox1->Controls->Add(this->checkHQ);
			this->groupBox1->Controls->Add(this->radioLossy);
			this->groupBox1->Controls->Add(this->radioLossless);
			this->groupBox1->Location = System::Drawing::Point(58, 268);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(326, 47);
			this->groupBox1->TabIndex = 7;
			this->groupBox1->TabStop = false;
			// 
			// linkLabel2
			// 
			this->linkLabel2->ActiveLinkColor = System::Drawing::Color::DarkRed;
			this->linkLabel2->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->linkLabel2->AutoSize = true;
			this->linkLabel2->BackColor = System::Drawing::Color::Transparent;
			this->linkLabel2->LinkColor = System::Drawing::Color::DarkRed;
			this->linkLabel2->Location = System::Drawing::Point(164, 78);
			this->linkLabel2->Name = L"linkLabel2";
			this->linkLabel2->Size = System::Drawing::Size(152, 13);
			this->linkLabel2->TabIndex = 1;
			this->linkLabel2->TabStop = true;
			this->linkLabel2->Text = L"cuj2k.project@googlemail.com";
			this->linkLabel2->VisitedLinkColor = System::Drawing::Color::DarkRed;
			this->linkLabel2->LinkClicked += gcnew System::Windows::Forms::LinkLabelLinkClickedEventHandler(this, &Form1::linkLabel2_LinkClicked);
			// 
			// label6
			// 
			this->label6->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->label6->BackColor = System::Drawing::Color::Transparent;
			this->label6->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(230)), static_cast<System::Int32>(static_cast<System::Byte>(0)), 
				static_cast<System::Int32>(static_cast<System::Byte>(0)));
			this->label6->Location = System::Drawing::Point(55, 20);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(261, 27);
			this->label6->TabIndex = 17;
			this->label6->Text = L"(c) 2009 Norbert Fürst, Martin Heide, Armin Weiß, Simon Papandreou, Ana Balevic";
			this->label6->TextAlign = System::Drawing::ContentAlignment::TopRight;
			// 
			// panel1
			// 
			this->panel1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->panel1->BackgroundImage = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"panel1.BackgroundImage")));
			this->panel1->Controls->Add(this->linkLabel2);
			this->panel1->Controls->Add(this->label6);
			this->panel1->Controls->Add(this->linkLabel1);
			this->panel1->Location = System::Drawing::Point(130, -11);
			this->panel1->Name = L"panel1";
			this->panel1->Size = System::Drawing::Size(333, 100);
			this->panel1->TabIndex = 18;
			// 
			// label7
			// 
			this->label7->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(70, 353);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(74, 13);
			this->label7->TabIndex = 19;
			this->label7->Text = L"Output format:";
			// 
			// comboOutFormat
			// 
			this->comboOutFormat->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->comboOutFormat->FormattingEnabled = true;
			this->comboOutFormat->Items->AddRange(gcnew cli::array< System::Object^  >(3) {L"JP2", L"J2K", L"MJ2"});
			this->comboOutFormat->Location = System::Drawing::Point(160, 347);
			this->comboOutFormat->Name = L"comboOutFormat";
			this->comboOutFormat->Size = System::Drawing::Size(75, 21);
			this->comboOutFormat->TabIndex = 20;
			this->comboOutFormat->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboOutFormat_SelectedIndexChanged);
			// 
			// label8
			// 
			this->label8->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(70, 369);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(293, 13);
			this->label8->TabIndex = 21;
			this->label8->Text = L"(JP2=normal JPEG2000 files;  MJ2=Motion JPEG2000 video)";
			// 
			// advancedBn
			// 
			this->advancedBn->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->advancedBn->Location = System::Drawing::Point(71, 430);
			this->advancedBn->Name = L"advancedBn";
			this->advancedBn->Size = System::Drawing::Size(115, 25);
			this->advancedBn->TabIndex = 22;
			this->advancedBn->Text = L"Advanced....";
			this->advancedBn->UseVisualStyleBackColor = true;
			this->advancedBn->Click += gcnew System::EventHandler(this, &Form1::advancedBn_Click);
			// 
			// resetBn
			// 
			this->resetBn->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->resetBn->Location = System::Drawing::Point(206, 430);
			this->resetBn->Name = L"resetBn";
			this->resetBn->Size = System::Drawing::Size(115, 25);
			this->resetBn->TabIndex = 23;
			this->resetBn->Text = L"Reset to default";
			this->resetBn->UseVisualStyleBackColor = true;
			this->resetBn->Click += gcnew System::EventHandler(this, &Form1::resetBn_Click);
			// 
			// label3
			// 
			this->label3->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(70, 395);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(79, 13);
			this->label3->TabIndex = 24;
			this->label3->Text = L"MJ2 output file:";
			// 
			// textMJ2Output
			// 
			this->textMJ2Output->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->textMJ2Output->Location = System::Drawing::Point(160, 392);
			this->textMJ2Output->Name = L"textMJ2Output";
			this->textMJ2Output->Size = System::Drawing::Size(208, 20);
			this->textMJ2Output->TabIndex = 25;
			// 
			// MJ2selectBn
			// 
			this->MJ2selectBn->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->MJ2selectBn->Location = System::Drawing::Point(374, 389);
			this->MJ2selectBn->Name = L"MJ2selectBn";
			this->MJ2selectBn->Size = System::Drawing::Size(71, 25);
			this->MJ2selectBn->TabIndex = 26;
			this->MJ2selectBn->Text = L"Select...";
			this->MJ2selectBn->UseVisualStyleBackColor = true;
			this->MJ2selectBn->Click += gcnew System::EventHandler(this, &Form1::MJ2selectBn_Click);
			// 
			// MJ2outputDlg
			// 
			this->MJ2outputDlg->DefaultExt = L"mj2";
			this->MJ2outputDlg->Filter = L"Motion JPEG2000 videos (*.mj2)|*.mj2|All files (*.*)|*.*";
			this->MJ2outputDlg->RestoreDirectory = true;
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(135)), static_cast<System::Int32>(static_cast<System::Byte>(174)), 
				static_cast<System::Int32>(static_cast<System::Byte>(197)));
			this->ClientSize = System::Drawing::Size(459, 534);
			this->Controls->Add(this->MJ2selectBn);
			this->Controls->Add(this->textMJ2Output);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->resetBn);
			this->Controls->Add(this->advancedBn);
			this->Controls->Add(this->label8);
			this->Controls->Add(this->comboOutFormat);
			this->Controls->Add(this->label7);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->panel1);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->textOutputSize);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->comboSizeMode);
			this->Controls->Add(this->encodeBn);
			this->Controls->Add(this->removeAllBn);
			this->Controls->Add(this->removeBn);
			this->Controls->Add(this->addBn);
			this->Controls->Add(this->inputfilesList);
			this->Controls->Add(this->label1);
			this->MinimumSize = System::Drawing::Size(467, 525);
			this->Name = L"Form1";
			this->Text = L"CUJ2K 1.1 - JPEG2000 encoder on CUDA";
			this->Load += gcnew System::EventHandler(this, &Form1::Form1_Load);
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &Form1::Form1_FormClosing);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->pictureBox1))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			this->panel1->ResumeLayout(false);
			this->panel1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	private:
		static const int MAJOR_VERSION = 1, MINOR_VERSION = 1;

		static const int SIZEMODE_MAX=0, SIZEMODE_PERCENT=1, 
			SIZEMODE_RATIO=2, SIZEMODE_BYTES=3, SIZEMODE_KB=4, SIZEMODE_MB=5;
		static const int CB_AUTO=0, CB_16=1, CB_32=2, CB_64=3;

		static String^ settingsFilename = 
			System::Environment::GetFolderPath(System::Environment::SpecialFolder::ApplicationData)
				+ L"\\cuj2k_1.1_dotnet_gui.ini";

		array<int ^> ^ deviceIndices;

		//bool checkForUpdates;

		AdvancedForm ^advForm;


		static String ^ getTempFilename(String ^ ext) {
			static int counter=0;

			counter++;
			return(IO::Path::GetTempPath() + L"\\cuj2k_temp_" + 
				Process::GetCurrentProcess()->Id + L"_" + counter + L"_" +
				DateTime::Now.Millisecond + ext);
		}

	private: 
		System::Void loadSettings() {
			IO::StreamReader ^fp = IO::File::OpenText(settingsFilename);
			try {
				radioLossless->Checked = bool::Parse(fp->ReadLine());
				radioLossy->Checked = bool::Parse(fp->ReadLine());
				checkHQ->Checked = bool::Parse(fp->ReadLine());
				textOutputSize->Text = fp->ReadLine();
				comboSizeMode->SelectedIndex = int::Parse(fp->ReadLine());
				comboOutFormat->SelectedIndex = int::Parse(fp->ReadLine());
				advForm->comboCBSize->SelectedIndex = int::Parse(fp->ReadLine());
				advForm->comboCUDAdevices->SelectedIndex = int::Parse(fp->ReadLine());
				advForm->checkStreaming->Checked = bool::Parse(fp->ReadLine());
				advForm->checkBenchmark->Checked = bool::Parse(fp->ReadLine());
				advForm->textAddOptions->Text = fp->ReadLine();
				advForm->checkUpdates->Checked = bool::Parse(fp->ReadLine());
			}
			finally {
				if(fp)
					delete (IDisposable^)fp;
			}
		}

	private: 
		System::Void saveSettings() {
			IO::StreamWriter ^fp = IO::File::CreateText(settingsFilename);
			try {
				fp->WriteLine(radioLossless->Checked.ToString());
				fp->WriteLine(radioLossy->Checked.ToString());
				fp->WriteLine(checkHQ->Checked.ToString());
				fp->WriteLine(textOutputSize->Text);
				fp->WriteLine(comboSizeMode->SelectedIndex.ToString());
				fp->WriteLine(comboOutFormat->SelectedIndex.ToString());
				fp->WriteLine(advForm->comboCBSize->SelectedIndex.ToString());
				fp->WriteLine(advForm->comboCUDAdevices->SelectedIndex.ToString());
				fp->WriteLine(advForm->checkStreaming->Checked.ToString());
				fp->WriteLine(advForm->checkBenchmark->Checked.ToString());
				fp->WriteLine(advForm->textAddOptions->Text);
				fp->WriteLine(advForm->checkUpdates->Checked.ToString());
			}
			finally {
				if(fp)
					delete (IDisposable^)fp;
			}
		}

	private: 
		//settings that are loaded if file could not be read
		System::Void defaultSettings() {
			radioLossless->Checked = true;
			radioLossy->Checked = false;
			checkHQ->Checked = false;
			textOutputSize->Text = L"";
			comboSizeMode->SelectedIndex = SIZEMODE_MAX;
			comboOutFormat->SelectedIndex = 0;
			advForm->comboCBSize->SelectedIndex = CB_AUTO;
			advForm->comboCUDAdevices->SelectedIndex = 0;
			advForm->checkStreaming->Checked = true;
			advForm->checkBenchmark->Checked = false;
			advForm->textAddOptions->Text = L"";
			advForm->checkUpdates->Checked = true;
		}


	private: 
		System::Void linkLabel1_LinkClicked(System::Object^  sender, System::Windows::Forms::LinkLabelLinkClickedEventArgs^  e) {
			Process::Start(L"http://cuj2k.sourceforge.net/");
		}
	private: 
		System::Void linkLabel2_LinkClicked(System::Object^  sender, System::Windows::Forms::LinkLabelLinkClickedEventArgs^  e) {
			Process::Start(L"mailto:cuj2k.project@googlemail.com");
		}

private: System::Void MJ2selectBn_Click(System::Object^  sender, System::EventArgs^  e) {
			 if(MJ2outputDlg->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
				 textMJ2Output->Text = MJ2outputDlg->FileName;
			 }
		 }

	private:
		System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
			if(this->inputfilesDlg->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
				for(int i = 0; i < this->inputfilesDlg->FileNames->Length; i++)
					this->inputfilesList->Items->Add(this->inputfilesDlg->FileNames[i]);
			}
		}

	private: 
		System::Void removeAllBn_Click(System::Object^  sender, System::EventArgs^  e) {
			this->inputfilesList->Items->Clear();
		}

	private: 
		System::Void removeSelectedFiles() {
			for(int i = this->inputfilesList->Items->Count - 1; i >= 0; i--) {
				if(this->inputfilesList->GetSelected(i))
					this->inputfilesList->Items->RemoveAt(i);
			}
		}

	private: 
		System::Void removeBn_Click(System::Object^  sender, System::EventArgs^  e) {
			this->removeSelectedFiles();
		}





	private:
		System::Void encodeBn_Click(System::Object^  sender, System::EventArgs^  e) {
			const int MAXLEN_CMDLINE = 1950;
			String ^exe;

			if(this->inputfilesList->Items->Count <= 0) {
				MessageBox::Show(L"Please select some input file(s) first!",
					L"", MessageBoxButtons::OK, MessageBoxIcon::Information);
				return;
			}
			
			exe = L"cuj2k";
			/*if(this->checkStreaming->Checked)
				exe = L"cuj2k";
			else
				exe = L"cuj2k_nostream";*/

			//first build options-commandline, it is the same for each launch
			bool lossy = this->radioLossy->Checked;
			bool hq = this->checkHQ->Checked;
			int sizeMode = this->comboSizeMode->SelectedIndex;

			StringBuilder^ cmdOptions = gcnew StringBuilder(L"", MAXLEN_CMDLINE+10);

			if(!(advForm->checkStreaming->Checked))
				cmdOptions->Append(L" -nostream");

			if(advForm->comboCUDAdevices->SelectedIndex > 0) {
				cmdOptions->Append(L" -setdev ");
				cmdOptions->Append(deviceIndices[advForm->comboCUDAdevices->SelectedIndex]);
			}
			/*if(this->radioDeviceUser->Checked && (this->textDeviceUser->Text->Length > 0)){
				cmdOptions->Append(L" -setdev ");
				cmdOptions->Append(this->textDeviceUser->Text);
			}*/
			if(advForm->checkBenchmark->Checked)
				cmdOptions->Append(L" -benchmark");

			cmdOptions->Append(L" ");
			cmdOptions->Append(advForm->textAddOptions->Text);

			bool mj2 = false;
			String ^mj2_basename;
			if(this->comboOutFormat->SelectedIndex == 1) //codestream only
				cmdOptions->Append(L" -format j2k");
			else if(this->comboOutFormat->SelectedIndex == 2) { //MJ2 
				mj2 = true;
				mj2_basename = textMJ2Output->Text;
				if(mj2_basename->Length == 0) {
					MessageBox::Show(L"Please specify an MJ2 output file.",
							L"Error", MessageBoxButtons::OK, MessageBoxIcon::Exclamation);
					return;						
				}

				if(mj2_basename->EndsWith(L".mj2", true /*ignore case*/,
					nullptr /*culture*/))
				{
					//cut off .mj2 since it is automatically appended
					//by mj2_wrapper
					mj2_basename = mj2_basename->Substring(0, mj2_basename->Length - 4);
				}
				cmdOptions->Append(L" -format j2k -o \"" + mj2_basename + L"_$05d.j2k\"");
				//cmdOptions->Append(L" -mj2 \"" + basename + L"\"");
			}
			switch(advForm->comboCBSize->SelectedIndex) {
				case CB_AUTO: break; //default for cuj2k
				case CB_16: cmdOptions->Append(L" -cb 16"); break;
				case CB_32: cmdOptions->Append(L" -cb 32"); break;
				case CB_64: cmdOptions->Append(L" -cb 64"); break;
			}

			if(lossy) {
				cmdOptions->Append(L" -irrev");
				if(hq)
					cmdOptions->Append(L" -hq");
			}// (-rev is default)

			//default is no PCRD, so don't put this on cmdline
			if((sizeMode != SIZEMODE_MAX) && (this->textOutputSize->Text->Length > 0)) {
				String ^strSize = 
					this->textOutputSize->Text->Replace(L',', L'.');
				if(sizeMode == SIZEMODE_PERCENT) {
					cmdOptions->Append(L" -sizerel ");
					try {
						//get percent value and divide by 100
						double ratio = Double::Parse(strSize, 
							CultureInfo::InvariantCulture) / 100.0;
						//InvariantCulture => use . and no ,
						cmdOptions->Append(ratio.ToString(CultureInfo::InvariantCulture));
					}
					catch(Exception^ e) {
						MessageBox::Show(L"Wrong number format for output filesize:\n" +
							e->Message + L"\n\nCorrect values are e.g. 4 or 10.5 .",
							L"Error", MessageBoxButtons::OK, MessageBoxIcon::Exclamation);
						return;						
					}
				}
				else if(sizeMode == SIZEMODE_RATIO) {
					cmdOptions->Append(L" -ratio ");
					cmdOptions->Append(strSize);
				}
				else { //absolute size
					cmdOptions->Append(L" -size ");
					cmdOptions->Append(strSize);
					if(sizeMode == SIZEMODE_KB)
						cmdOptions->Append(L"K");
					else if(sizeMode == SIZEMODE_MB)
						cmdOptions->Append(L"M");
				}
			}

			String ^ batFilename = getTempFilename(L".bat");
			IO::StreamWriter ^batFile = IO::File::CreateText(batFilename);

			int file_i = 0;
			//outer loop: until all files are processed
			while(file_i < this->inputfilesList->Items->Count) {
				//put options in front of commandline
				StringBuilder^ cmdArgs = gcnew StringBuilder(cmdOptions->ToString(),
					MAXLEN_CMDLINE+10);

				//make sure that file counter is correct if multiple
				//cuj2k calls are needed for MJ2 creation
				if(mj2)
					cmdArgs->Append(L" -starti " + file_i);

				//inner loop: until all files are processed or commandline is too long
				while(file_i < this->inputfilesList->Items->Count) {
					String ^filename = safe_cast<String^> (this->inputfilesList->Items[file_i]);
					if(exe->Length + cmdArgs->Length + filename->Length + 3 > MAXLEN_CMDLINE)
						break;

					cmdArgs->Append(L" \"");
					cmdArgs->Append(filename);
					cmdArgs->Append(L"\"");
					file_i++;
				}
				batFile->WriteLine(exe + L" " + cmdArgs->ToString());

				/*try {
					Process ^process = Process::Start(exe, cmdArgs->ToString());
					process->WaitForExit();
				}
				catch(Exception ^e) {
					MessageBox::Show(L"Could not execute " + exe + L".exe:\n" +
						e->Message + L"\n\nNote: The files cuj2k.exe and cuj2k_nostream.exe must reside\n" +
						L"in the same directory as this GUI program.",
						L"Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return;						
				}*/
			}

			if(mj2)
				batFile->WriteLine(L"mj2_wrapper \"" + mj2_basename + "\" \"" + 
					mj2_basename + ".mj2\"");

			batFile->WriteLine(L"@pause");
			if(batFile)
				delete (IDisposable^)batFile;

			Process ^process = Process::Start(batFilename);
			process->WaitForExit();

			IO::File::Delete(batFilename);
		}

		void updateCheckThread() {
			//check for new version
			try 
			{
				HttpHelper^ versionFile = gcnew HttpHelper(L"http://cuj2k.sourceforge.net/latest-version.txt");
				int major = int::Parse(versionFile->getValue(L"latest-major-version"));
				int minor = int::Parse(versionFile->getValue(L"latest-minor-version"));
				String ^changes = versionFile->getValue(L"changelist");
				String ^download = versionFile->getValue(L"download-page");

				if(changes != nullptr && download != nullptr && 
					((major > MAJOR_VERSION) || 
					 (major == MAJOR_VERSION && minor > MINOR_VERSION)))
				{
					UpdateForm ^upForm = gcnew UpdateForm();
					upForm->labelVersion->Text = major + L"." + minor;

					array<Char>^chars = {'\n'};
					upForm->textChanges->Lines = changes->Split( chars );

					upForm->downloadPage = download;
					//upForm->bnYes->Focus();
					upForm->ShowDialog(); //waits until form is closed/invisible
					advForm->checkUpdates->Checked = !(upForm->checkDontAsk->Checked);
					upForm->Close();
					//this->textAddOptions->Text = L"fertig!";
					//Boolean ^b = new Boolean(true);
					
				}
			}
			catch(Exception ^) { }
		}

	private: 
		System::Void Form1_Load(System::Object^  sender, System::EventArgs^  e) {
			advForm = gcnew AdvancedForm();

			cudaError_t cuda_err;

			int deviceCount;
			cuda_err = cudaGetDeviceCount(&deviceCount);
			if(cuda_err == cudaSuccess) {
				deviceIndices = gcnew array<int ^>(deviceCount+1);
				//MessageBox::Show(L"# of devices: " + deviceCount.ToString());

				//index #0 is "auto-select" => start with 1
				int indexInList=1;

				for(int i = 0; i < deviceCount; i++) {
					cudaDeviceProp deviceProp;
					cuda_err = cudaGetDeviceProperties(&deviceProp, i);
					if(cuda_err != cudaSuccess)
						break;
					//only choose devices with compute cap. >= 1.1
					if((deviceProp.major > 1) ||
						((deviceProp.major == 1) && (deviceProp.minor >= 1))) 
					{
						deviceIndices[indexInList++] = i;
						advForm->comboCUDAdevices->Items->Add
							(L"#" + i + L": " + gcnew String(deviceProp.name) /* + 
							L" (compute cap.: " + deviceProp.major + L"." + deviceProp.minor +
							L" RAM: " + deviceProp.totalGlobalMem + L" MB)" */ );
					}
					//this->comboSizeMode->Items->Add(gcnew String(deviceProp.name));
				}
			}
			if(cuda_err != cudaSuccess) {
				MessageBox::Show(L"Error occured when enumerating CUDA devices:\n\n" +
					gcnew String(cudaGetErrorString(cuda_err)) + 
					L"\n\nTry using 'auto-select' for CUDA device.",
					L"CUDA Error", MessageBoxButtons::OK, MessageBoxIcon::Error);			
			}

			try {
				loadSettings();
			}
			catch(Exception^ ) {
				//MessageBox::Show(e->ToString());
				defaultSettings();
			}

			if(advForm->checkUpdates->Checked) {
				ThreadStart^ job = gcnew ThreadStart(this, &dotnet_gui::Form1::updateCheckThread);
				Thread^ thread = gcnew Thread(job);
				thread->Start();
			}
		}

	private: 
		System::Void comboSizeMode_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			this->textOutputSize->Enabled = 
				(this->comboSizeMode->SelectedIndex != 0);
		}

	private: 
		System::Void radioLossless_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			this->checkHQ->Enabled = this->radioLossy->Checked;
			if(this->radioLossless->Checked)
				this->comboSizeMode->SelectedIndex = SIZEMODE_MAX;
		}

	private: 
		System::Void radioLossy_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			this->checkHQ->Enabled = this->radioLossy->Checked;
			if(this->radioLossless->Checked)
				this->comboSizeMode->SelectedIndex = SIZEMODE_MAX;
		}


private: System::Void inputfilesList_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
			 if(e->KeyCode == Keys::Delete)
				 this->removeSelectedFiles();
		 }
private: System::Void Form1_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e) {
			 try {
				 saveSettings();
			 }
			 catch(Exception^ ){
				 //MessageBox::Show(e->ToString());
			 }
		 }

private: System::Void advancedBn_Click(System::Object^  sender, System::EventArgs^  e) {
			 if(!advForm->Visible)
				 advForm->Show();
			 advForm->BringToFront();
		 }
private: System::Void comboOutFormat_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			 if(comboOutFormat->SelectedIndex == 2) {
				 label3->Enabled = true;
				 textMJ2Output->Enabled = true;
				 MJ2selectBn->Enabled = true;
			 }
			 else {
				 label3->Enabled = false;
				 textMJ2Output->Enabled = false;
				 MJ2selectBn->Enabled = false;
			 }
		 }
private: System::Void resetBn_Click(System::Object^  sender, System::EventArgs^  e) {
			 defaultSettings();
		 }
};
}

